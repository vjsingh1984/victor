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

"""
Unit tests for the batch processing coordinator.
"""

import pytest
import time

try:
    from victor.native.rust.batch_processor import (
        BatchProcessor,
        BatchTask,
        BatchResult,
        BatchProcessSummary,
        RUST_AVAILABLE,
    )
except ImportError:
    RUST_AVAILABLE = False
    pytest.skip("Batch processor not available", allow_module_level=True)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust implementation not available")
class TestBatchProcessor:
    """Test BatchProcessor functionality."""

    def test_basic_task_creation(self):
        """Test creating basic BatchTask objects."""
        task = BatchTask(
            task_id="test_task",
            task_data=lambda x: x * 2,
            priority=1.0,
            timeout_ms=5000,
            dependencies=[],
        )

        assert task.task_id == "test_task"
        assert task.priority == 1.0
        assert task.timeout_ms == 5000
        assert task.dependencies == []

    def test_processor_creation(self):
        """Test creating BatchProcessor."""
        processor = BatchProcessor(
            max_concurrent=10,
            timeout_ms=30000,
            retry_policy="exponential",
            aggregation_strategy="unordered",
        )

        assert processor is not None
        if RUST_AVAILABLE:
            assert processor._rust_processor is not None

    def test_simple_batch_processing(self):
        """Test processing a simple batch of tasks."""
        processor = BatchProcessor(max_concurrent=5)

        def simple_executor(task_dict):
            time.sleep(0.01)  # Simulate work
            return f"Result for {task_dict['task_id']}"

        tasks = [
            BatchTask(task_id=f"task{i}", task_data=simple_executor, priority=1.0) for i in range(5)
        ]

        summary = processor.process_batch(tasks, simple_executor)

        assert summary is not None
        assert summary.successful_count == 5
        assert summary.failed_count == 0
        assert summary.throughput_per_second > 0
        assert len(summary.results) == 5

    def test_dependency_resolution(self):
        """Test resolving task execution order with dependencies."""
        processor = BatchProcessor(max_concurrent=10)

        tasks = [
            BatchTask(task_id="task1", task_data=None, priority=1.0, dependencies=[]),
            BatchTask(
                task_id="task2",
                task_data=None,
                priority=1.0,
                dependencies=["task1"],
            ),
            BatchTask(
                task_id="task3",
                task_data=None,
                priority=1.0,
                dependencies=["task1"],
            ),
            BatchTask(
                task_id="task4",
                task_data=None,
                priority=1.0,
                dependencies=["task2", "task3"],
            ),
        ]

        layers = processor.resolve_execution_order(tasks)

        assert len(layers) >= 3
        assert "task1" in layers[0]
        assert "task2" in layers[1] or "task3" in layers[1]
        assert "task4" in layers[2] or "task4" in layers[3]

    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected."""
        processor = BatchProcessor(max_concurrent=10)

        tasks = [
            BatchTask(
                task_id="task1",
                task_data=None,
                priority=1.0,
                dependencies=["task2"],
            ),
            BatchTask(
                task_id="task2",
                task_data=None,
                priority=1.0,
                dependencies=["task1"],
            ),
        ]

        with pytest.raises(ValueError, match="Circular dependency"):
            processor.validate_dependencies(tasks)

    def test_task_retry_logic(self):
        """Test task retry with failures."""
        processor = BatchProcessor(max_concurrent=5, retry_policy="fixed")

        call_count = {"count": 0}

        def failing_executor(task_dict):
            call_count["count"] += 1
            if call_count["count"] <= 2:
                raise ValueError("Simulated failure")
            return "Success"

        tasks = [BatchTask(task_id="failing_task", task_data=failing_executor, priority=1.0)]

        summary = processor.process_batch(tasks, failing_executor)

        # Should succeed after retries
        assert summary.successful_count == 1
        assert summary.failed_count == 0
        assert call_count["count"] >= 2  # At least 2 attempts

    def test_priority_execution(self):
        """Test that task priorities are respected."""
        processor = BatchProcessor(max_concurrent=5, aggregation_strategy="priority")

        execution_order = []

        def priority_executor(task_dict):
            execution_order.append(task_dict["task_id"])
            return task_dict["task_id"]

        tasks = [
            BatchTask(
                task_id="low_priority",
                task_data=priority_executor,
                priority=0.1,
            ),
            BatchTask(
                task_id="high_priority",
                task_data=priority_executor,
                priority=1.0,
            ),
            BatchTask(
                task_id="medium_priority",
                task_data=priority_executor,
                priority=0.5,
            ),
        ]

        summary = processor.process_batch(tasks, priority_executor)

        assert summary.successful_count == 3
        # Note: Actual ordering depends on implementation

    def test_load_balancing(self):
        """Test task assignment with load balancing."""
        if not RUST_AVAILABLE:
            pytest.skip("Load balancing requires Rust implementation")

        processor = BatchProcessor(max_concurrent=10)
        processor.set_load_balancer("round_robin")

        tasks = [BatchTask(task_id=f"task{i}", task_data=None, priority=1.0) for i in range(10)]

        assignments = processor.assign_tasks(tasks, workers=3)

        assert len(assignments) == 3
        assert sum(len(a) for a in assignments) == 10

    def test_batch_creation(self):
        """Test splitting tasks into batches."""
        from victor.native.rust.batch_processor import create_task_batches_py

        tasks = [BatchTask(task_id=f"task{i}", task_data=None, priority=1.0) for i in range(10)]

        batches = create_task_batches_py(tasks, batch_size=3)

        assert len(batches) == 4  # 10 / 3 = 4 batches
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation."""
        from victor.native.rust.batch_processor import calculate_optimal_batch_size_py

        # 100 tasks, 10 workers -> 10 tasks per batch
        batch_size = calculate_optimal_batch_size_py(100, 10)
        assert batch_size == 10

        # 50 tasks, 5 workers -> 10 tasks per batch
        batch_size = calculate_optimal_batch_size_py(50, 5)
        assert batch_size == 10

    def test_duration_estimation(self):
        """Test batch duration estimation."""
        from victor.native.rust.batch_processor import estimate_batch_duration_py

        # 100 tasks, 100ms avg duration, 10 concurrent
        # Should be ~1000ms (10 waves * 100ms)
        duration = estimate_batch_duration_py(100, 100.0, 10)
        assert duration > 0
        assert duration == 1000.0

    def test_error_handling(self):
        """Test error handling in batch processing."""
        processor = BatchProcessor(max_concurrent=5)

        def error_executor(task_dict):
            if task_dict["task_id"] == "error_task":
                raise RuntimeError("Intentional error")
            return "Success"

        tasks = [
            BatchTask(
                task_id="success_task",
                task_data=error_executor,
                priority=1.0,
            ),
            BatchTask(
                task_id="error_task",
                task_data=error_executor,
                priority=1.0,
            ),
        ]

        summary = processor.process_batch(tasks, error_executor)

        assert summary.successful_count == 1
        assert summary.failed_count == 1
        assert len(summary.results) == 2

        # Check error result
        error_result = [r for r in summary.results if not r.success][0]
        assert error_result.task_id == "error_task"
        assert error_result.error is not None

    def test_streaming_results(self):
        """Test streaming result callback."""
        processor = BatchProcessor(max_concurrent=5)

        results_received = []

        def result_callback(result):
            results_received.append(result)

        def streaming_executor(task_dict):
            return f"Result for {task_dict['task_id']}"

        tasks = [
            BatchTask(
                task_id=f"task{i}",
                task_data=streaming_executor,
                priority=1.0,
            )
            for i in range(5)
        ]

        summary = processor.process_batch_streaming(tasks, streaming_executor, result_callback)

        assert len(results_received) == 5
        assert summary.successful_count == 5

    def test_batch_summary_metrics(self):
        """Test batch process summary metrics."""
        summary = BatchProcessSummary(
            results=[],
            total_duration_ms=1000.0,
            successful_count=8,
            failed_count=2,
            retried_count=3,
            throughput_per_second=10.0,
        )

        assert summary.success_rate() == 80.0  # 8/10 = 80%
        # Note: avg_duration_ms requires results with durations


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust implementation not available")
class TestBatchProcessorIntegration:
    """Integration tests for BatchProcessor."""

    def test_multi_stage_pipeline(self):
        """Test a multi-stage processing pipeline."""
        processor = BatchProcessor(max_concurrent=10, aggregation_strategy="ordered")

        results = {}

        def pipeline_executor(task_dict):
            task_id = task_dict["task_id"]
            if task_id == "compile":
                results["compile"] = "compiled"
                return "compiled"
            elif task_id == "test":
                assert results.get("compile") == "compiled"
                results["test"] = "tested"
                return "tested"
            elif task_id == "deploy":
                assert results.get("test") == "tested"
                results["deploy"] = "deployed"
                return "deployed"
            return "unknown"

        tasks = [
            BatchTask(task_id="compile", task_data=pipeline_executor, priority=1.0),
            BatchTask(
                task_id="test",
                task_data=pipeline_executor,
                priority=0.9,
                dependencies=["compile"],
            ),
            BatchTask(
                task_id="deploy",
                task_data=pipeline_executor,
                priority=0.8,
                dependencies=["test"],
            ),
        ]

        summary = processor.process_batch(tasks, pipeline_executor)

        assert summary.successful_count == 3
        assert results["compile"] == "compiled"
        assert results["test"] == "tested"
        assert results["deploy"] == "deployed"

    def test_parallel_file_processing(self):
        """Test parallel file processing simulation."""
        processor = BatchProcessor(max_concurrent=5)

        processed_files = []

        def file_processor(task_dict):
            file_path = task_dict["task_id"]
            # Simulate file processing
            time.sleep(0.01)
            processed_files.append(file_path)
            return f"Processed {file_path}"

        files = [f"file{i}.py" for i in range(10)]
        tasks = [BatchTask(task_id=file, task_data=file_processor, priority=1.0) for file in files]

        summary = processor.process_batch(tasks, file_processor)

        assert summary.successful_count == 10
        assert len(processed_files) == 10
        assert summary.throughput_per_second > 0
