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

"""Tests for evaluation harness module."""

import pytest
from pathlib import Path

from victor.evaluation.harness import (
    BaseBenchmarkRunner,
    TaskEnvironment,
    EvaluationHarness,
)
from victor.evaluation.protocol import (
    BenchmarkTask,
    BenchmarkType,
    EvaluationConfig,
    TaskResult,
    TaskStatus,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


class MockBenchmarkRunner(BaseBenchmarkRunner):
    """Mock runner for testing."""

    def __init__(self, tasks: list[BenchmarkTask] | None = None):
        self._tasks = tasks or []
        self._run_results = {}

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.HUMAN_EVAL

    async def load_tasks(self, config: EvaluationConfig) -> list[BenchmarkTask]:
        return self._filter_tasks(self._tasks, config)

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        if task.task_id in self._run_results:
            return self._run_results[task.task_id]
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.PASSED,
            generated_code=agent_output,
        )

    def set_run_result(self, task_id: str, result: TaskResult):
        """Set the result for a specific task."""
        self._run_results[task_id] = result


def create_test_task(
    task_id: str = "test_task_1",
    language: str = "python",
    category: str = "test",
    difficulty: str = "easy",
) -> BenchmarkTask:
    """Create a test task."""
    return BenchmarkTask(
        task_id=task_id,
        benchmark=BenchmarkType.HUMAN_EVAL,
        description="Test task",
        language=language,
        prompt="Write a function",
        category=category,
        difficulty=difficulty,
    )


def create_test_config(
    benchmark: BenchmarkType = BenchmarkType.HUMAN_EVAL,
    model: str = "test-model",
    max_tasks: int | None = None,
    parallel_tasks: int = 1,
    task_ids: list[str] | None = None,
    languages: list[str] | None = None,
    categories: list[str] | None = None,
    difficulties: list[str] | None = None,
) -> EvaluationConfig:
    """Create a test evaluation config."""
    return EvaluationConfig(
        benchmark=benchmark,
        model=model,
        max_tasks=max_tasks,
        parallel_tasks=parallel_tasks,
        task_ids=task_ids,
        languages=languages,
        categories=categories,
        difficulties=difficulties,
    )


# =============================================================================
# BASE BENCHMARK RUNNER TESTS
# =============================================================================


class TestBaseBenchmarkRunner:
    """Tests for BaseBenchmarkRunner filtering."""

    def test_filter_tasks_no_filters(self):
        """Test filtering with no filters returns all tasks."""
        tasks = [
            create_test_task("task1"),
            create_test_task("task2"),
            create_test_task("task3"),
        ]
        runner = MockBenchmarkRunner(tasks)
        config = create_test_config()

        filtered = runner._filter_tasks(tasks, config)

        assert len(filtered) == 3

    def test_filter_tasks_by_task_ids(self):
        """Test filtering by specific task IDs."""
        tasks = [
            create_test_task("task1"),
            create_test_task("task2"),
            create_test_task("task3"),
        ]
        runner = MockBenchmarkRunner(tasks)
        config = create_test_config(task_ids=["task1", "task3"])

        filtered = runner._filter_tasks(tasks, config)

        assert len(filtered) == 2
        assert filtered[0].task_id == "task1"
        assert filtered[1].task_id == "task3"

    def test_filter_tasks_by_language(self):
        """Test filtering by language."""
        tasks = [
            create_test_task("task1", language="python"),
            create_test_task("task2", language="javascript"),
            create_test_task("task3", language="python"),
        ]
        runner = MockBenchmarkRunner(tasks)
        config = create_test_config(languages=["python"])

        filtered = runner._filter_tasks(tasks, config)

        assert len(filtered) == 2

    def test_filter_tasks_by_category(self):
        """Test filtering by category."""
        tasks = [
            create_test_task("task1", category="algorithms"),
            create_test_task("task2", category="strings"),
            create_test_task("task3", category="algorithms"),
        ]
        runner = MockBenchmarkRunner(tasks)
        config = create_test_config(categories=["algorithms"])

        filtered = runner._filter_tasks(tasks, config)

        assert len(filtered) == 2

    def test_filter_tasks_by_difficulty(self):
        """Test filtering by difficulty."""
        tasks = [
            create_test_task("task1", difficulty="easy"),
            create_test_task("task2", difficulty="hard"),
            create_test_task("task3", difficulty="easy"),
        ]
        runner = MockBenchmarkRunner(tasks)
        config = create_test_config(difficulties=["easy"])

        filtered = runner._filter_tasks(tasks, config)

        assert len(filtered) == 2

    def test_filter_tasks_with_max_tasks(self):
        """Test filtering with max_tasks limit."""
        tasks = [create_test_task(f"task{i}") for i in range(10)]
        runner = MockBenchmarkRunner(tasks)
        config = create_test_config(max_tasks=5)

        filtered = runner._filter_tasks(tasks, config)

        assert len(filtered) == 5

    def test_filter_tasks_combined_filters(self):
        """Test combining multiple filters."""
        tasks = [
            create_test_task("task1", language="python", difficulty="easy"),
            create_test_task("task2", language="python", difficulty="hard"),
            create_test_task("task3", language="javascript", difficulty="easy"),
            create_test_task("task4", language="python", difficulty="easy"),
        ]
        runner = MockBenchmarkRunner(tasks)
        config = create_test_config(languages=["python"], difficulties=["easy"], max_tasks=1)

        filtered = runner._filter_tasks(tasks, config)

        assert len(filtered) == 1
        assert filtered[0].task_id == "task1"


# =============================================================================
# TASK ENVIRONMENT TESTS
# =============================================================================


class TestTaskEnvironment:
    """Tests for TaskEnvironment class."""

    def test_init_default(self):
        """Test default initialization."""
        task = create_test_task()
        env = TaskEnvironment(task)

        assert env.task is task
        assert env.use_docker is False
        assert env._temp_dir is None

    def test_init_with_options(self):
        """Test initialization with options."""
        task = create_test_task()
        workspace = Path("/tmp/test")
        env = TaskEnvironment(
            task, workspace_dir=workspace, use_docker=True, docker_image="python:3.12"
        )

        assert env.workspace_dir == workspace
        assert env.use_docker is True
        assert env.docker_image == "python:3.12"

    @pytest.mark.asyncio
    async def test_setup_creates_temp_dir(self):
        """Test setup creates temporary directory."""
        task = create_test_task(task_id="test_task")
        env = TaskEnvironment(task)

        try:
            workspace = await env.setup()
            assert workspace.exists()
            assert "eval_test_task_" in str(workspace)
        finally:
            await env.cleanup()

    @pytest.mark.asyncio
    async def test_setup_sanitizes_task_id(self):
        """Test setup sanitizes task_id with special characters."""
        task = create_test_task(task_id="repo/test/task")
        env = TaskEnvironment(task)

        try:
            workspace = await env.setup()
            assert workspace.exists()
            # Should have replaced / with _
            assert "/" not in workspace.name
        finally:
            await env.cleanup()

    @pytest.mark.asyncio
    async def test_setup_writes_context_code(self):
        """Test setup writes context code to file."""
        task = BenchmarkTask(
            task_id="test",
            benchmark=BenchmarkType.HUMAN_EVAL,
            description="Test",
            language="python",
            prompt="Test",
            context_code="def hello(): return 'world'",
        )
        env = TaskEnvironment(task)

        try:
            workspace = await env.setup()
            solution_file = workspace / "solution.py"
            assert solution_file.exists()
            assert "def hello()" in solution_file.read_text()
        finally:
            await env.cleanup()

    @pytest.mark.asyncio
    async def test_setup_writes_test_code(self):
        """Test setup writes test code to file."""
        task = BenchmarkTask(
            task_id="test",
            benchmark=BenchmarkType.HUMAN_EVAL,
            description="Test",
            language="python",
            prompt="Test",
            test_code="def test_hello(): assert hello() == 'world'",
        )
        env = TaskEnvironment(task)

        try:
            workspace = await env.setup()
            test_file = workspace / "test_solution.py"
            assert test_file.exists()
            assert "test_hello" in test_file.read_text()
        finally:
            await env.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_removes_temp_dir(self):
        """Test cleanup removes temporary directory."""
        task = create_test_task()
        env = TaskEnvironment(task)

        workspace = await env.setup()
        assert workspace.exists()

        await env.cleanup()
        assert not workspace.exists()

    @pytest.mark.asyncio
    async def test_cleanup_handles_already_deleted(self):
        """Test cleanup handles already deleted directory."""
        task = create_test_task()
        env = TaskEnvironment(task)

        await env.setup()
        # Delete manually
        if env._temp_dir:
            import shutil

            shutil.rmtree(env._temp_dir)

        # Should not raise
        await env.cleanup()

    @pytest.mark.asyncio
    async def test_apply_patch_without_setup(self):
        """Test apply_patch returns False without setup."""
        task = create_test_task()
        env = TaskEnvironment(task)

        result = await env.apply_patch("some patch")

        assert result is False

    @pytest.mark.asyncio
    async def test_run_tests_without_setup(self):
        """Test run_tests returns error without setup."""
        task = create_test_task()
        env = TaskEnvironment(task)

        passed, total, stdout, stderr = await env.run_tests()

        assert passed == 0
        assert total == 0
        assert "Environment not set up" in stderr

    def test_parse_test_output_pytest_format(self):
        """Test parsing pytest output format."""
        task = create_test_task()
        env = TaskEnvironment(task)

        output = "===== 5 passed, 2 failed in 1.23s ====="
        passed, total = env._parse_test_output(output)

        assert passed == 5
        assert total == 7

    def test_parse_test_output_pytest_only_passed(self):
        """Test parsing pytest output with only passed."""
        task = create_test_task()
        env = TaskEnvironment(task)

        output = "===== 10 passed in 0.5s ====="
        passed, total = env._parse_test_output(output)

        assert passed == 10
        assert total == 10

    def test_parse_test_output_unittest_format(self):
        """Test parsing unittest output format."""
        task = create_test_task()
        env = TaskEnvironment(task)

        output = """Ran 8 tests in 0.001s
OK (failures=1, errors=2)"""
        passed, total = env._parse_test_output(output)

        assert passed == 5
        assert total == 8

    def test_parse_test_output_unknown_format(self):
        """Test parsing unknown output format returns zeros."""
        task = create_test_task()
        env = TaskEnvironment(task)

        output = "Some random output"
        passed, total = env._parse_test_output(output)

        assert passed == 0
        assert total == 0


# =============================================================================
# EVALUATION HARNESS TESTS
# =============================================================================


class TestEvaluationHarness:
    """Tests for EvaluationHarness class."""

    def test_init_default(self):
        """Test default initialization."""
        harness = EvaluationHarness()

        assert len(harness._runners) == 0
        assert harness._results_dir.exists()

    def test_init_with_runners(self):
        """Test initialization with runners."""
        runner = MockBenchmarkRunner()
        harness = EvaluationHarness(runners={BenchmarkType.HUMAN_EVAL: runner})

        assert len(harness._runners) == 1

    def test_register_runner(self):
        """Test registering a runner."""
        harness = EvaluationHarness()
        runner = MockBenchmarkRunner()

        harness.register_runner(runner)

        assert harness.get_runner(BenchmarkType.HUMAN_EVAL) is runner

    def test_get_runner_not_found(self):
        """Test getting non-existent runner returns None."""
        harness = EvaluationHarness()

        runner = harness.get_runner(BenchmarkType.SWE_BENCH)

        assert runner is None

    @pytest.mark.asyncio
    async def test_run_evaluation_no_runner(self):
        """Test run_evaluation raises without runner."""
        harness = EvaluationHarness()
        config = create_test_config(benchmark=BenchmarkType.SWE_BENCH)

        async def dummy_callback(task):
            return "code"

        with pytest.raises(ValueError, match="No runner for benchmark"):
            await harness.run_evaluation(config, dummy_callback)

    @pytest.mark.asyncio
    async def test_run_evaluation_sequential(self):
        """Test sequential evaluation run."""
        tasks = [create_test_task("task1"), create_test_task("task2")]
        runner = MockBenchmarkRunner(tasks)
        harness = EvaluationHarness(runners={BenchmarkType.HUMAN_EVAL: runner})
        config = create_test_config(parallel_tasks=1)

        async def agent_callback(task):
            return f"def {task.task_id}(): pass"

        result = await harness.run_evaluation(config, agent_callback)

        assert len(result.task_results) == 2
        assert result.start_time is not None
        assert result.end_time is not None

    @pytest.mark.asyncio
    async def test_run_evaluation_with_progress_callback(self):
        """Test evaluation calls progress callback."""
        tasks = [create_test_task("task1")]
        runner = MockBenchmarkRunner(tasks)
        harness = EvaluationHarness(runners={BenchmarkType.HUMAN_EVAL: runner})
        config = create_test_config()

        progress_called = []

        def progress_callback(idx, total, result):
            progress_called.append((idx, total, result.task_id))

        async def agent_callback(task):
            return "code"

        await harness.run_evaluation(config, agent_callback, progress_callback)

        assert len(progress_called) == 1
        assert progress_called[0][0] == 0
        assert progress_called[0][1] == 1

    @pytest.mark.asyncio
    async def test_run_evaluation_parallel(self):
        """Test parallel evaluation run."""
        tasks = [create_test_task(f"task{i}") for i in range(5)]
        runner = MockBenchmarkRunner(tasks)
        harness = EvaluationHarness(runners={BenchmarkType.HUMAN_EVAL: runner})
        config = create_test_config(parallel_tasks=3)

        async def agent_callback(task):
            return "code"

        result = await harness.run_evaluation(config, agent_callback)

        assert len(result.task_results) == 5

    @pytest.mark.asyncio
    async def test_run_evaluation_handles_errors(self):
        """Test evaluation handles task errors gracefully."""
        tasks = [create_test_task("task1")]
        runner = MockBenchmarkRunner(tasks)
        harness = EvaluationHarness(runners={BenchmarkType.HUMAN_EVAL: runner})
        config = create_test_config()

        async def error_callback(task):
            raise ValueError("Agent error")

        result = await harness.run_evaluation(config, error_callback)

        assert len(result.task_results) == 1
        assert result.task_results[0].status == TaskStatus.ERROR
        assert "Agent error" in result.task_results[0].error_message

    @pytest.mark.asyncio
    async def test_run_evaluation_with_failing_task(self):
        """Test evaluation with failing task result."""
        tasks = [create_test_task("task1")]
        runner = MockBenchmarkRunner(tasks)
        runner.set_run_result(
            "task1",
            TaskResult(task_id="task1", status=TaskStatus.FAILED, error_message="Test failed"),
        )
        harness = EvaluationHarness(runners={BenchmarkType.HUMAN_EVAL: runner})
        config = create_test_config()

        async def agent_callback(task):
            return "code"

        result = await harness.run_evaluation(config, agent_callback)

        assert result.task_results[0].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_run_evaluation_with_max_tasks(self):
        """Test evaluation respects max_tasks."""
        tasks = [create_test_task(f"task{i}") for i in range(10)]
        runner = MockBenchmarkRunner(tasks)
        harness = EvaluationHarness(runners={BenchmarkType.HUMAN_EVAL: runner})
        config = create_test_config(max_tasks=3)

        async def agent_callback(task):
            return "code"

        result = await harness.run_evaluation(config, agent_callback)

        assert len(result.task_results) == 3


# =============================================================================
# INTEGRATION-LIKE TESTS
# =============================================================================


class TestTaskEnvironmentIntegration:
    """Integration tests for TaskEnvironment."""

    @pytest.mark.asyncio
    async def test_full_workflow_without_repo(self):
        """Test full workflow without repo cloning."""
        task = BenchmarkTask(
            task_id="integration_test",
            benchmark=BenchmarkType.HUMAN_EVAL,
            description="Test function",
            language="python",
            prompt="Write hello function",
            context_code="def hello(): return 'world'",
            test_code="assert hello() == 'world'",
        )
        env = TaskEnvironment(task)

        try:
            workspace = await env.setup()

            # Verify files exist
            assert (workspace / "solution.py").exists()
            assert (workspace / "test_solution.py").exists()

            # Verify content
            solution = (workspace / "solution.py").read_text()
            assert "def hello()" in solution

        finally:
            await env.cleanup()
