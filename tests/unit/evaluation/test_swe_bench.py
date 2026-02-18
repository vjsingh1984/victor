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

"""Unit tests for benchmark runners.

Tests for SWE-bench, HumanEval, and MBPP benchmark runners.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from victor.evaluation.benchmarks.swe_bench import (
    SWEBenchRunner,
    HumanEvalRunner,
    MBPPRunner,
)
from victor.evaluation.protocol import BenchmarkType, EvaluationConfig, TaskStatus


# =============================================================================
# SWEBenchRunner Tests
# =============================================================================


class TestSWEBenchRunnerInit:
    """Tests for SWEBenchRunner initialization."""

    def test_init_with_defaults(self):
        """Initialize with default parameters."""
        runner = SWEBenchRunner()
        assert runner._split == "test"
        assert runner._dataset_path is None
        assert runner._tasks_cache is None

    def test_init_with_custom_params(self):
        """Initialize with custom parameters."""
        runner = SWEBenchRunner(dataset_path=Path("/data/swe.jsonl"), split="dev")
        assert runner._split == "dev"
        assert runner._dataset_path == Path("/data/swe.jsonl")

    def test_benchmark_type_property(self):
        """Return correct benchmark type."""
        runner = SWEBenchRunner()
        assert runner.benchmark_type == BenchmarkType.SWE_BENCH


class TestExtractPatch:
    """Tests for _extract_patch method."""

    def test_extract_standard_diff_patch(self):
        """Extract standard diff patch format."""
        runner = SWEBenchRunner()

        output = """Some text
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,1 +1,2 @@
-old line
+new line
"""

        patch = runner._extract_patch(output)
        assert "diff --git a/file.py b/file.py" in patch
        assert "--- a/file.py" in patch
        assert "+new line" in patch

    def test_extract_patch_from_code_block(self):
        """Extract patch from markdown code block."""
        runner = SWEBenchRunner()

        output = """Here's the fix:

```diff
--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-old
+new
```
"""

        patch = runner._extract_patch(output)
        assert "--- a/file.py" in patch
        assert "+new" in patch

    def test_extract_patch_no_diff_returns_empty(self):
        """Return empty string when no diff found."""
        runner = SWEBenchRunner()

        output = "Just regular text without any diff"

        patch = runner._extract_patch(output)
        assert patch == ""

    def test_extract_patch_with_multiple_diffs(self):
        """Extract first complete diff from multiple diffs."""
        runner = SWEBenchRunner()

        output = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,1 +1,1 @@
-a
+b
diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,1 +1,1 @@
-c
+d
"""

        patch = runner._extract_patch(output)
        # Should extract at least the first diff
        assert "diff --git a/file1.py" in patch


class TestSWEBenchRunnerLoadTasks:
    """Tests for load_tasks method."""

    @pytest.mark.asyncio
    async def test_load_tasks_returns_cached_tasks(self):
        """Load tasks returns cached tasks when available."""
        runner = SWEBenchRunner()
        runner._tasks_cache = []
        config = EvaluationConfig(
            benchmark=BenchmarkType.SWE_BENCH, model="test", max_tasks=10
        )

        tasks = await runner.load_tasks(config)
        assert tasks == []

    @pytest.mark.asyncio
    async def test_load_tasks_filters_by_config(self):
        """Load tasks filters based on config max_tasks."""
        from victor.evaluation.protocol import BenchmarkTask

        mock_task1 = BenchmarkTask(
            task_id="task-1",
            benchmark=BenchmarkType.SWE_BENCH,
            description="Task 1",
        )
        mock_task2 = BenchmarkTask(
            task_id="task-2",
            benchmark=BenchmarkType.SWE_BENCH,
            description="Task 2",
        )

        runner = SWEBenchRunner()
        runner._tasks_cache = [mock_task1, mock_task2]
        config = EvaluationConfig(
            benchmark=BenchmarkType.SWE_BENCH, model="test", max_tasks=1
        )

        tasks = await runner.load_tasks(config)
        assert len(tasks) == 1
        assert tasks[0].task_id == "task-1"


# =============================================================================
# HumanEvalRunner Tests
# =============================================================================


class TestHumanEvalRunnerInit:
    """Tests for HumanEvalRunner initialization."""

    def test_init_default(self):
        """HumanEvalRunner uses default initialization."""
        runner = HumanEvalRunner()
        # No special attributes since it uses BaseBenchmarkRunner defaults
        assert runner.benchmark_type == BenchmarkType.HUMAN_EVAL

    def test_benchmark_type_property(self):
        """Return correct benchmark type."""
        runner = HumanEvalRunner()
        assert runner.benchmark_type == BenchmarkType.HUMAN_EVAL


# =============================================================================
# MBPPRunner Tests
# =============================================================================


class TestMBPPRunnerInit:
    """Tests for MBPPRunner initialization."""

    def test_init_with_default_split(self):
        """Initialize with default split."""
        runner = MBPPRunner()
        assert runner._split == "test"

    def test_init_with_custom_split(self):
        """Initialize with custom split."""
        runner = MBPPRunner(split="dev")
        assert runner._split == "dev"

    def test_benchmark_type_property(self):
        """Return correct benchmark type."""
        runner = MBPPRunner()
        assert runner.benchmark_type == BenchmarkType.MBPP


class TestMBPPBuildPrompt:
    """Tests for _build_prompt method."""

    def test_build_prompt_from_mbpp_item(self):
        """Build prompt from MBPP item."""
        runner = MBPPRunner()

        item = {
            "task_id": 1,
            "text": "Write a function to add two numbers.",
            "code": "",
            "test_list": ["assert add(1, 2) == 3"],
        }

        prompt = runner._build_prompt(item)
        assert "Write a function to add two numbers." in prompt


class TestMBPPBuildTestCode:
    """Tests for _build_test_code method."""

    def test_build_test_code_from_mbpp_item(self):
        """Build test code from MBPP item."""
        runner = MBPPRunner()

        item = {
            "task_id": 1,
            "text": "Write a function.",
            "code": "",
            "test_list": [
                "assert add(1, 2) == 3",
                "assert add(0, 0) == 0",
            ],
        }

        test_code = runner._build_test_code(item)
        assert "assert add(1, 2) == 3" in test_code
        assert "assert add(0, 0) == 0" in test_code

    def test_build_test_code_with_setup(self):
        """Build test code with setup code."""
        runner = MBPPRunner()

        item = {
            "task_id": 1,
            "text": "Write a function.",
            "code": "",
            "test_setup_code": "def add(a, b): return a + b",
            "test_list": ["assert add(1, 2) == 3"],
        }

        test_code = runner._build_test_code(item)
        assert "def add(a, b): return a + b" in test_code
        assert "assert add(1, 2) == 3" in test_code


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestRunTaskErrorHandling:
    """Tests for run_task error handling."""

    @pytest.mark.asyncio
    async def test_run_task_handles_exception(self):
        """Run task properly handles exceptions."""
        from victor.evaluation.protocol import EvaluationConfig

        runner = SWEBenchRunner()
        task = MagicMock()
        task.task_id = "test-1"
        task.description = "Test task"
        task.test_code = "assert True"

        agent_output = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old\n+new"
        config = EvaluationConfig(
            benchmark=BenchmarkType.SWE_BENCH, model="test", use_docker=False
        )

        # Mock TaskEnvironment to raise exception on setup
        with patch("victor.evaluation.benchmarks.swe_bench.TaskEnvironment") as MockEnv:
            mock_env_instance = MagicMock()
            mock_env_instance.setup = AsyncMock(side_effect=Exception("Setup failed"))
            mock_env_instance.cleanup = AsyncMock()
            MockEnv.return_value = mock_env_instance

            result = await runner.run_task(task, agent_output, config)

            assert result.status == TaskStatus.ERROR
            assert "Setup failed" in result.error_message


# =============================================================================
# Configuration Tests
# =============================================================================


class TestBenchmarkConfiguration:
    """Tests for benchmark configuration and filtering."""

    def test_swe_bench_config_default_path(self):
        """SWEBenchRunner defaults to no dataset_path."""
        runner = SWEBenchRunner()
        assert runner._dataset_path is None

    def test_mbpp_config_default_split(self):
        """MBPPRunner defaults to 'test' split."""
        runner = MBPPRunner()
        assert runner._split == "test"

    def test_all_runners_have_correct_benchmark_type(self):
        """All runners report correct benchmark type."""
        swe_runner = SWEBenchRunner()
        human_runner = HumanEvalRunner()
        mbpp_runner = MBPPRunner()

        assert swe_runner.benchmark_type == BenchmarkType.SWE_BENCH
        assert human_runner.benchmark_type == BenchmarkType.HUMAN_EVAL
        assert mbpp_runner.benchmark_type == BenchmarkType.MBPP
