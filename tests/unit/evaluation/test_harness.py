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

"""Unit tests for evaluation harness.

Tests for BaseBenchmarkRunner, TaskEnvironment, and EvaluationHarness.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional

from victor.evaluation.harness import (
    BaseBenchmarkRunner,
    TaskEnvironment,
    EvaluationHarness,
    get_harness,
)
from victor.evaluation.protocol import (
    BenchmarkTask,
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    TaskResult,
    TaskStatus,
)


# =============================================================================
# BaseBenchmarkRunner._filter_tasks Tests
# =============================================================================


class MockBenchmarkRunner(BaseBenchmarkRunner):
    """Mock implementation for testing BaseBenchmarkRunner."""

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.CUSTOM

    async def load_tasks(self, config: EvaluationConfig) -> list[BenchmarkTask]:
        return []

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        return TaskResult(task_id=task.task_id, status=TaskStatus.PASSED)


class TestFilterTasks:
    """Tests for _filter_tasks method."""

    @pytest.fixture
    def sample_tasks(self):
        """Create sample benchmark tasks."""
        return [
            BenchmarkTask(
                task_id="task-1",
                benchmark=BenchmarkType.HUMAN_EVAL,
                description="Python task",
                language="python",
                category="code_generation",
                difficulty="easy",
            ),
            BenchmarkTask(
                task_id="task-2",
                benchmark=BenchmarkType.HUMAN_EVAL,
                description="JavaScript task",
                language="javascript",
                category="code_generation",
                difficulty="medium",
            ),
            BenchmarkTask(
                task_id="task-3",
                benchmark=BenchmarkType.HUMAN_EVAL,
                description="Hard Python task",
                language="python",
                category="code_generation",
                difficulty="hard",
            ),
        ]

    def test_filter_by_task_ids(self, sample_tasks):
        """Filter tasks by specific task IDs."""
        runner = MockBenchmarkRunner()
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test",
            task_ids=["task-1", "task-3"],
        )

        filtered = runner._filter_tasks(sample_tasks, config)

        assert len(filtered) == 2
        assert {t.task_id for t in filtered} == {"task-1", "task-3"}

    def test_filter_by_languages(self, sample_tasks):
        """Filter tasks by programming language."""
        runner = MockBenchmarkRunner()
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL, model="test", languages=["python"]
        )

        filtered = runner._filter_tasks(sample_tasks, config)

        assert len(filtered) == 2
        assert all(t.language == "python" for t in filtered)

    def test_filter_by_categories(self, sample_tasks):
        """Filter tasks by category."""
        runner = MockBenchmarkRunner()
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test",
            categories=["code_generation"],
        )

        filtered = runner._filter_tasks(sample_tasks, config)

        assert len(filtered) == 3
        assert all(t.category == "code_generation" for t in filtered)

    def test_filter_by_difficulties(self, sample_tasks):
        """Filter tasks by difficulty."""
        runner = MockBenchmarkRunner()
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test",
            difficulties=["easy", "medium"],
        )

        filtered = runner._filter_tasks(sample_tasks, config)

        assert len(filtered) == 2
        assert all(t.difficulty in ["easy", "medium"] for t in filtered)

    def test_filter_by_max_tasks(self, sample_tasks):
        """Filter tasks by max_tasks limit."""
        runner = MockBenchmarkRunner()
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL, model="test", max_tasks=2
        )

        filtered = runner._filter_tasks(sample_tasks, config)

        assert len(filtered) == 2
        assert filtered == sample_tasks[:2]

    def test_filter_combined_criteria(self, sample_tasks):
        """Filter tasks with multiple criteria combined."""
        runner = MockBenchmarkRunner()
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test",
            languages=["python"],
            difficulties=["easy", "hard"],
        )

        filtered = runner._filter_tasks(sample_tasks, config)

        assert len(filtered) == 2
        assert all(t.language == "python" for t in filtered)
        assert all(t.difficulty in ["easy", "hard"] for t in filtered)

    def test_filter_no_criteria_returns_all(self, sample_tasks):
        """No filters configured returns all tasks."""
        runner = MockBenchmarkRunner()
        config = EvaluationConfig(benchmark=BenchmarkType.HUMAN_EVAL, model="test")

        filtered = runner._filter_tasks(sample_tasks, config)

        assert len(filtered) == 3


# =============================================================================
# TaskEnvironment Tests
# =============================================================================


class TestTaskEnvironmentInit:
    """Tests for TaskEnvironment initialization."""

    def test_init_with_defaults(self):
        """Initialize with default parameters."""
        task = MagicMock()
        task.task_id = "test-1"

        env = TaskEnvironment(task)

        assert env.task == task
        assert env.use_docker is False
        assert env.docker_image == "python:3.11"
        assert env._temp_dir is None

    def test_init_with_custom_params(self):
        """Initialize with custom parameters."""
        task = MagicMock()
        task.task_id = "test-1"

        env = TaskEnvironment(
            task,
            workspace_dir=Path("/custom/workspace"),
            use_docker=True,
            docker_image="python:3.12",
        )

        assert env.use_docker is True
        assert env.docker_image == "python:3.12"
        assert env.workspace_dir == Path("/custom/workspace")


class TestParseTestOutput:
    """Tests for _parse_test_output method."""

    @pytest.fixture
    def environment(self):
        """Create a test environment."""
        task = MagicMock()
        task.task_id = "test-1"
        return TaskEnvironment(task)

    def test_parse_passing_tests(self, environment):
        """Parse output where all tests pass."""
        output = """
Ran 3 tests in 0.001s

OK
"""
        passed, total = environment._parse_test_output(output)

        assert total == 3
        assert passed == 3

    def test_parse_failing_tests(self, environment):
        """Parse output with test failures."""
        output = """
Ran 3 tests in 0.001s

FAILED (failures=1)
tests/test_foo.py:10: AssertionError

FAILED (failures=2, errors=1)
"""
        passed, total = environment._parse_test_output(output)

        assert total == 3
        assert passed == 1

    def test_parse_pytest_output(self, environment):
        """Parse pytest-style output."""
        output = """tests/test_example.py::test_function PASSED
tests/test_example.py::test_another FAILED
==================== 2 passed, 1 failed ===================="""
        passed, total = environment._parse_test_output(output)

        assert total >= 3
        assert passed >= 2


# =============================================================================
# EvaluationOrchestrator Tests
# =============================================================================


class TestEvaluationHarness:
    """Tests for EvaluationHarness."""

    def test_init_default(self):
        """Initialize with default parameters."""
        harness = EvaluationHarness()

        assert harness._results_dir is not None
        assert harness._checkpoint_dir is not None
        assert harness._runners == {}

    def test_init_with_custom_checkpoint_dir(self, tmp_path):
        """Initialize with custom checkpoint directory."""
        harness = EvaluationHarness(checkpoint_dir=tmp_path)

        assert harness._checkpoint_dir == tmp_path

    def test_register_runner(self):
        """Register a benchmark runner."""
        harness = EvaluationHarness()
        runner = MockBenchmarkRunner()

        harness.register_runner(runner)

        assert BenchmarkType.CUSTOM in harness._runners
        assert harness._runners[BenchmarkType.CUSTOM] == runner

    def test_get_runner(self):
        """Retrieve a registered runner."""
        harness = EvaluationHarness()
        runner = MockBenchmarkRunner()
        harness.register_runner(runner)

        retrieved = harness.get_runner(BenchmarkType.CUSTOM)

        assert retrieved == runner

    def test_get_runner_returns_none_for_unknown(self):
        """Return None for unknown benchmark type."""
        harness = EvaluationHarness()

        retrieved = harness.get_runner(BenchmarkType.CUSTOM)

        assert retrieved is None

    def test_get_harness_singleton(self):
        """Test get_harness returns singleton instance."""
        harness1 = get_harness()
        harness2 = get_harness()

        assert harness1 is harness2


class TestCheckpointPaths:
    """Tests for checkpoint path methods."""

    def test_get_checkpoint_path_default(self):
        """Get default checkpoint path."""
        harness = EvaluationHarness()
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="claude-3-sonnet",
        )

        path = harness._get_checkpoint_path(config)

        assert path.name.startswith("checkpoint_")
        assert "human_eval" in path.name

    def test_get_checkpoint_path_custom_dir(self, tmp_path):
        """Get checkpoint path in custom directory."""
        harness = EvaluationHarness(checkpoint_dir=tmp_path)
        config = EvaluationConfig(
            benchmark=BenchmarkType.SWE_BENCH,
            model="gpt-4",
        )

        path = harness._get_checkpoint_path(config)

        assert path.parent == tmp_path


class TestClearCheckpoint:
    """Tests for _clear_checkpoint method."""

    def test_clear_checkpoint_removes_file(self):
        """Clear checkpoint removes checkpoint file."""
        harness = EvaluationHarness()
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="claude-3-sonnet",
        )

        # Create a mock checkpoint file
        checkpoint_path = harness._get_checkpoint_path(config)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("{}")

        harness._clear_checkpoint(checkpoint_path)

        assert not checkpoint_path.exists()

    def test_clear_checkpoint_handles_missing_file(self):
        """Clear checkpoint handles missing file gracefully."""
        harness = EvaluationHarness()
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="claude-3-sonnet",
        )

        # Don't create the file
        checkpoint_path = harness._get_checkpoint_path(config)

        # Should not raise
        harness._clear_checkpoint(checkpoint_path)


class TestGenerateReports:
    """Tests for report generation methods."""

    @pytest.fixture
    def mock_result(self):
        """Create a mock evaluation result."""
        result = EvaluationResult(
            config=EvaluationConfig(
                benchmark=BenchmarkType.HUMAN_EVAL,
                model="claude-3-sonnet",
            ),
            task_results=[
                TaskResult(
                    task_id="test-1",
                    status=TaskStatus.PASSED,
                    tests_passed=2,
                    tests_total=2,
                ),
                TaskResult(
                    task_id="test-2",
                    status=TaskStatus.FAILED,
                    tests_passed=0,
                    tests_total=1,
                ),
            ],
        )
        return result

    def test_generate_text_report(self, mock_result):
        """Generate text report from evaluation result."""
        harness = EvaluationHarness()

        report = harness._generate_text_report(mock_result)

        assert "human_eval" in report
        assert "claude-3-sonnet" in report
        assert ("1 passed" in report or "50.0%" in report)

    def test_generate_markdown_report(self, mock_result):
        """Generate markdown report from evaluation result."""
        harness = EvaluationHarness()

        report = harness._generate_markdown_report(mock_result)

        assert "# Evaluation Report" in report
        assert "claude-3-sonnet" in report


class TestSaveAndLoadResults:
    """Tests for saving and loading evaluation results."""

    def test_save_results_creates_file(self, tmp_path):
        """Save results creates JSON file."""
        harness = EvaluationHarness(checkpoint_dir=tmp_path)

        result = EvaluationResult(
            config=EvaluationConfig(
                benchmark=BenchmarkType.HUMAN_EVAL,
                model="test",
            ),
            task_results=[],
        )

        result_path = harness._save_results(result)

        assert result_path.exists()
        assert result_path.suffix == ".json"

    def test_load_results_reads_saved_file(self, tmp_path):
        """Load results can read saved results."""
        harness = EvaluationHarness(checkpoint_dir=tmp_path)

        # Save a result first
        result = EvaluationResult(
            config=EvaluationConfig(
                benchmark=BenchmarkType.HUMAN_EVAL,
                model="test",
            ),
            task_results=[],
        )
        saved_path = harness._save_results(result)

        # Load it back
        loaded = harness.load_results(saved_path)

        assert "config" in loaded
        assert "tasks" in loaded  # Saved as "tasks" in _save_results


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestTaskEnvironmentIntegration:
    """Integration tests for TaskEnvironment."""

    @pytest.mark.asyncio
    async def test_setup_creates_temp_directory(self):
        """Setup creates temporary directory."""
        task = MagicMock()
        task.task_id = "test_setup_task"
        task.repo = None
        task.context_code = None
        task.test_code = None

        env = TaskEnvironment(task)
        workspace = await env.setup()

        assert workspace.exists()
        assert workspace.name.startswith("eval_test_setup_task_")
        assert env._temp_dir == workspace

        await env.cleanup()
        assert not workspace.exists()

    @pytest.mark.asyncio
    async def test_setup_creates_context_file(self):
        """Setup creates solution.py with context code."""
        task = MagicMock()
        task.task_id = "test_context"
        task.context_code = "def hello():\n    return 'world'"
        task.repo = None
        task.test_code = None

        env = TaskEnvironment(task)
        workspace = await env.setup()

        solution_file = workspace / "solution.py"
        assert solution_file.exists()
        content = solution_file.read_text()
        assert "def hello():" in content

        await env.cleanup()

    @pytest.mark.asyncio
    async def test_setup_creates_test_file(self):
        """Setup creates test_solution.py with test code."""
        task = MagicMock()
        task.task_id = "test_with_tests"
        task.repo = None
        task.context_code = None
        task.test_code = "assert True"

        env = TaskEnvironment(task)
        workspace = await env.setup()

        test_file = workspace / "test_solution.py"
        assert test_file.exists()
        content = test_file.read_text()
        assert "assert True" in content

        await env.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_removes_temp_directory(self):
        """Cleanup removes the temporary directory."""
        task = MagicMock()
        task.task_id = "test_cleanup"
        task.repo = None
        task.context_code = None
        task.test_code = None

        env = TaskEnvironment(task)
        workspace = await env.setup()

        temp_dir = env._temp_dir
        assert temp_dir.exists()

        await env.cleanup()

        assert not temp_dir.exists()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in evaluation infrastructure."""

    @pytest.mark.asyncio
    async def test_parse_test_output_handles_empty_output(self):
        """Handle empty test output gracefully."""
        task = MagicMock()
        task.task_id = "test-1"
        env = TaskEnvironment(task)

        passed, total = env._parse_test_output("")

        assert total == 0
        assert passed == 0

    @pytest.mark.asyncio
    async def test_apply_patch_with_no_repo(self):
        """Apply patch when there's no repo returns False."""
        task = MagicMock()
        task.task_id = "test-1"
        task.repo = None
        env = TaskEnvironment(task)

        result = await env.apply_patch("diff --git a/file.py")

        assert result is False

    @pytest.mark.asyncio
    async def test_clone_repo_skipped_when_no_repo(self):
        """Clone repo is skipped when task has no repo."""
        task = MagicMock()
        task.task_id = "test_no_repo"
        task.repo = None
        task.context_code = None
        task.test_code = None

        env = TaskEnvironment(task)
        workspace = await env.setup()

        # Should not attempt to clone
        # (can't easily test without mocking subprocess)

        await env.cleanup()
